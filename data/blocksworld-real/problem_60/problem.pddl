(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        orange - block
        blue - block
        yellow - block
        purple - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable orange)
        (on blue orange)
        (on yellow blue)
        (on purple yellow)
        (on green purple)
        (clear green)
        ; red
        ; orange -> blue -> yellow -> purple -> green
    )
    (:goal (and (on red green)
        (on purple yellow)
        (on blue purple)
        (on orange blue)))
)
    